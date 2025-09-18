# -*- coding: utf-8 -*-
import os
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio


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
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("PX4Env")


def _euler_to_quaternion(euler):
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

def _quaternion_to_euler(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    phi = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    theta = np.arcsin(2.0 * (w * y - z * x))
    psi = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([phi, theta, psi])

class QuadcopterEnv(gym.Env):
    """
    PX4四旋翼仿真环境用于强化学习

    一个Gym兼容的环境，用于训练四旋翼控制任务的RL智能体。
    基于PX4控制算法和动力学。
    """
    def __init__(self,
                 control_type: str = "xyz_pos",
                 trajectory_type: int = 5,
                 yaw_type: int = 3,
                 max_episode_steps: int = 2000,     # 每个回合的最大步数（以RL步计）
                 dt: float = 0.005,                  # 仿真积分时间步长（动力学/控制）
                 rl_dt: float = 0.05,                  # RL决策时间步长（每次step保持action的时长）
                 target_pos: Optional[np.ndarray] = None,
                 dense_reward_scale: float = 1.0,   # 即时奖励权重缩放
                 wind_type: str = 'None',
                 wind_strength: float = 2.0,
                 test_env_mode: bool = False,      # 测试模式：在 step 末尾写 CSV
                 render_mode: Optional[str] = None 
            ):

        """
        初始化四旋翼环境

        参数:
            control_type: 控制类型 ("xyz_pos", "xy_vel_z_pos", "xyz_vel")
            trajectory_type: 轨迹选择 (0-13)
            yaw_type: 偏航控制类型 (0-4)
            max_episode_steps: 每个回合的最大步数（以RL步计）
            rl_dt: RL 决策步长（秒），一个step保持action的时长
            dt: 仿真时间步长（秒）
            target_pos: 目标位置 [x, y, z]
            dense_reward_scale: 即时奖励缩放，便于调节奖励尺度
            wind_type: 风模型类型
            wind_strength: 风力强度
            render_mode: 渲染模式 ('human' 或 'rgb_array')
            test_env_mode: 当为 True 时，step 末尾会将 info 写入 CSV 以便分析
        """
        super(QuadcopterEnv, self).__init__()
        # 环境参数
        self.control_type = control_type
        self.trajectory_type = trajectory_type
        self.yaw_type = yaw_type
        self.max_episode_steps = max_episode_steps
        self.rl_dt = rl_dt
        self.dt = dt
        self.target_pos = target_pos if target_pos is not None else np.array([2.0, 2.0, 1.0])
        self.wind_type = wind_type
        self.wind_strength = wind_strength
        self.render_mode = render_mode
        self.dense_reward_scale = float(dense_reward_scale)
        self.test_env_mode = test_env_mode

        # 新增：渲染模式
        self.render_mode = render_mode
        self._render_inited = False
        self._fig = None
        self._ax = None

        # 奖励权重（即时）
        self.reward_weights = dict(
            w_pos_xy=0.3,
            w_pos_z=0.25,
            w_vel_xy=0.15,
            w_vel_z=0.1,
            w_att=0.1,
            w_omega=0.02,
        )

        # Segment 权重（本步内样本的RMSE/均值聚合）
        self.segment_weights = dict(
            W_pos_xy=0.3,
            W_pos_z=0.25,
            W_vel_xy=0.15,
            W_vel_z=0.1,
            W_att_roll=0.1,
            W_att_pitch=0.1,
            W_omega=0.02,

            # 暂时不开启平滑性奖励
            W_u=0.0,            # 0.05
            W_dv=0.0,           # 0.06
            W_domega=0.0,       # 0.07
            W_du=0.0,           # 0.08

            W_sat=0.15,
            W_over=0.15
        )

        # 限制与判据
        self.limits = dict(
            w_cmd_min=0.0,
            w_cmd_max=2e4,
            overshoot_z_tol=0.2,
            normalize_rate_by_dt=True  # 平滑性指标按dt归一
        )

        # 失败惩罚
        self.failure_penalty = -10.0  # 非正常截断的惩罚

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

        # 步级日志与 segment 统计
        self.step_log_rows = []             # 记录每个step的奖励构成，用于csv输出
        self.last_segment_metrics = None    # 最近一次窗口指标（本实现为“本步聚合”，此字段用于info兼容）

        # CSV 记录相关（仅 test_env_mode 时启用）
        self._csv_dir = None
        self._csv_path = None
        self._csv_fh = None
        self._csv_writer = None
        self._csv_header_written = False
        self.render_data = None
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
        self.stage = 0          # 初始课程学习的阶段

        # 子步越界保护标记
        self._force_truncate = False
    # =========================
    # CSV 工具
    # =========================
    def _open_step_csv(self):
        """当 test_env_mode=True 时，在每次 reset 打开一个新的 CSV 文件并写表头"""
        if not self.test_env_mode:
            return
        try:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._csv_dir = os.path.join("log_step_metrics", f"episode_{stamp}")
            os.makedirs(self._csv_dir, exist_ok=True)
            self._csv_path = os.path.join(self._csv_dir, "step_metrics.csv")
            self._csv_fh = open(self._csv_path, mode="w", newline="")
            self._csv_writer = csv.writer(self._csv_fh)
            # 写表头
            header = [
                # 基本步信息
                "step", "time_end", "executed_substeps",
                "dense_mean", "segment_reward", "total_reward", "dense_scale",
                # segment metrics
                "segment_duration", "rmse_pos_xy", "rmse_pos_z", "rmse_vel_xy", "rmse_vel_z",
                "rmse_att_roll", "rmse_att_pitch", "rms_omega",
                "mean_control_effort", "rms_dv", "rms_domega", "rms_du",
                "saturation_hits", "overshoot_hits",
                # 额外观测 & 误差
                "pos_x", "pos_y", "pos_z",
                "vel_x", "vel_y", "vel_z",
                "att_roll", "att_pitch", "att_yaw",
                "omega_p", "omega_q", "omega_r",
                "pos_ref_x", "pos_ref_y", "pos_ref_z",
                "vel_ref_x", "vel_ref_y", "vel_ref_z",
                "att_ref_roll", "att_ref_pitch", "att_ref_yaw",
                "omega_ref_p", "omega_ref_q", "omega_ref_r",
                "position_error", "episode_reward"
            ]
            self._csv_writer.writerow(header)
            self._csv_header_written = True
        except Exception as e:
            log.error(f"[step-csv] Open CSV failed: {e}")
            self._csv_fh = None
            self._csv_writer = None
            self._csv_header_written = False

    def _write_step_csv(self, info: Dict[str, Any]):
        """将当前 step 的 info 写入 CSV（仅 test_env_mode=True 时）"""
        if not self.test_env_mode or self._csv_writer is None or not self._csv_header_written:
            return
        try:
            br = info.get("step_reward_breakdown", {})
            last_seg = info.get("last_segment", {})  # 若 step_reward_breakdown 已含 segment，可用 br 覆盖
            # 观测信息
            pos = info.get("position", np.zeros(3, dtype=float))
            vel = info.get("velocity", np.zeros(3, dtype=float))
            att = info.get("attitude", np.zeros(3, dtype=float))
            omg = info.get("angularvelocity", np.zeros(3, dtype=float))
            pos_ref = info.get("pos_ref", np.zeros(3, dtype=float))
            vel_ref = info.get("vel_ref", np.zeros(3, dtype=float))
            att_ref = info.get("att_ref", np.zeros(3, dtype=float))
            omg_ref = info.get("angularvel_ref", np.zeros(3, dtype=float))

            row = [
                int(br.get("step", self.current_step)),
                float(br.get("time_end", self.time)),
                int(br.get("executed_substeps", 0)),
                float(br.get("dense_mean", 0.0)),
                float(br.get("segment_reward", 0.0)),
                float(br.get("total_reward", 0.0)),
                float(br.get("dense_scale", self.dense_reward_scale)),

                float(br.get("segment_duration", last_seg.get("segment_duration", 0.0))),
                float(br.get("rmse_pos_xy", last_seg.get("rmse_pos_xy", 0.0))),
                float(br.get("rmse_pos_z", last_seg.get("rmse_pos_z", 0.0))),
                float(br.get("rmse_vel_xy", last_seg.get("rmse_vel_xy", 0.0))),
                float(br.get("rmse_vel_z", last_seg.get("rmse_vel_z", 0.0))),
                float(br.get("rmse_att_roll", last_seg.get("rmse_att_roll", 0.0))),
                float(br.get("rmse_att_pitch", last_seg.get("rmse_att_pitch", 0.0))),
                float(br.get("rms_omega", last_seg.get("rms_omega", 0.0))),
                float(br.get("mean_control_effort", last_seg.get("mean_control_effort", 0.0))),
                float(br.get("rms_dv", last_seg.get("rms_dv", 0.0))),
                float(br.get("rms_domega", last_seg.get("rms_domega", 0.0))),
                float(br.get("rms_du", last_seg.get("rms_du", 0.0))),
                float(br.get("saturation_hits", last_seg.get("saturation_hits", 0.0))),
                float(br.get("overshoot_hits", last_seg.get("overshoot_hits", 0.0))),

                float(pos[0]), float(pos[1]), float(pos[2]),
                float(vel[0]), float(vel[1]), float(vel[2]),
                float(att[0]), float(att[1]), float(att[2]),
                float(omg[0]), float(omg[1]), float(omg[2]),
                float(pos_ref[0]), float(pos_ref[1]), float(pos_ref[2]),
                float(vel_ref[0]), float(vel_ref[1]), float(vel_ref[2]),
                float(att_ref[0]), float(att_ref[1]), float(att_ref[2]),
                float(omg_ref[0]), float(omg_ref[1]), float(omg_ref[2]),
                float(info.get("position_error", 0.0)),
                float(info.get("episode_reward", self.episode_reward)),
            ]
            self._csv_writer.writerow(row)
            # 及时 flush，便于长跑中观察
            self._csv_fh.flush()
        except Exception as e:
            log.error(f"[step-csv] Write CSV failed: {e}")

    def _close_step_csv(self):
        """关闭 CSV 文件"""
        try:
            if self._csv_fh is not None:
                self._csv_fh.flush()
                self._csv_fh.close()
        finally:
            self._csv_fh = None
            self._csv_writer = None
            self._csv_header_written = False
            self._csv_dir = None
            self._csv_path = None

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
        self.step_log_rows = []  # 清空步级日志
        self.last_segment_metrics = None

        if options is not None and 'target_pos' in options:
            self.target_pos = np.array(options['target_pos'])

        self._initialize_simulation()

        # 如果是测试环境模式，打开新的 CSV 文件并写表头
        self._close_step_csv()
        self._open_step_csv()

        if options and options.get('random_init', False):
            noise = np.random.normal(0, 0.1, 3)
            self.quad.pos += noise
            self.quad.state[0:3] = self.quad.pos

        obs = self._get_observation(self.ctrl.sDesCalc)
        info = self._get_info()

        self.step_action = np.zeros((20,), dtype=getattr(self.action_space, "dtype", np.float32))
        self.step_reward = 0.0
        self.render_data = self._pack_render_data(obs, self.step_action, self.step_reward)

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一个环境步骤：保持该action在rl_dt时长内，包含多个仿真子步"""
        self.step_action = action
        action = np.clip(action, 0.0, 1.0)          # Action参数剪切处理
        self._apply_pid_parameters(action)          # 将参数写入PID控制器

        try:
            import torch
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
        except Exception:
            pass

        # 需要运行的仿真子步数量
        substeps = max(1, int(round(self.rl_dt / self.dt)))     # 计算每个Step 需要多少个仿真子步
        step_reward_dense_sum = 0.0                             # 聚合该RL step内的即时奖励（dense）

        step_samples: List[Dict[str, float]] = []
        prev_cache = {"vel": None, "omega": None, "w_cmd": None}
        executed_substeps = 0  # 本步实际执行的子步数（用于更精确均值化） 

        terminated = False
        truncated = False

        # 在每个仿真步
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
            executed_substeps += 1

            # 计算即时状态误差
            sDesclc = self.ctrl.sDesCalc
            m = self._extract_metrics(sDesclc)

            # 即时奖励（dense，细粒度，引导稳定）
            dense_r = self._dense_reward_instant(m)
            step_reward_dense_sum += dense_r

            # 收集本步样本
            sample = self._collect_step_sample(m, prev_cache)
            step_samples.append(sample)

            # 过程内快速越界检查（如希望提前退出，可在此判定）
            if self._is_out_of_bounds():
                truncated = True
                break

        # RL步计数
        self.current_step += 1

        # step奖励 = 平均即时dense + 分段奖励（本步内聚合）
        # 即时奖励
        dense_mean = step_reward_dense_sum / max(1, executed_substeps)

        # 本步分段奖励
        segment_reward, metrics = self._compute_segment_reward_step(step_samples)
        self.last_segment_metrics = metrics if metrics else self.last_segment_metrics
        
        # 合成当步总奖励
        reward = dense_mean + segment_reward  

        # 终止/截断判定
        # 1) 达到最大步数 -> terminated
        if self.current_step >= self.max_episode_steps or self._force_truncate:
            terminated = True

        # 2) 非法姿态/位置越界 -> truncated
        if not terminated and self._is_out_of_bounds():
            truncated = True

        # 失败惩罚
        if truncated and not terminated:
            reward += self.failure_penalty

        self.step_reward = reward
        self.episode_reward += reward

        # 结束时记录可用 episode 长度
        if terminated or truncated:
            max_substeps_total = self.episode_t_all.shape[0]
            self._episode_len = min(self.current_substep, max_substeps_total)

        obs = self._get_observation(self.ctrl.sDesCalc)
        info = self._get_info()

        # 在 info 中回传每步奖励拆解，供分析
        info["step_reward_breakdown"] = {
            "step": int(self.current_step),
            "time_end": float(self.time),
            "executed_substeps": int(executed_substeps),
            "dense_mean": float(dense_mean),
            "segment_reward": float(segment_reward),
            "total_reward": float(reward),
            "dense_scale": float(self.dense_reward_scale),
        }
        if metrics is not None:
            info["step_reward_breakdown"].update({
                "segment_duration": metrics.get("segment_duration", 0.0),
                "rmse_pos_xy": metrics.get("rmse_pos_xy", 0.0),
                "rmse_pos_z": metrics.get("rmse_pos_z", 0.0),
                "rmse_vel_xy": metrics.get("rmse_vel_xy", 0.0),
                "rmse_vel_z": metrics.get("rmse_vel_z", 0.0),
                "rmse_att_roll": metrics.get("rmse_att_roll", 0.0),
                "rmse_att_pitch": metrics.get("rmse_att_pitch", 0.0),
                "rms_omega": metrics.get("rms_omega", 0.0),
                "mean_control_effort": metrics.get("mean_control_effort", 0.0),
                "rms_dv": metrics.get("rms_dv", 0.0),
                "rms_domega": metrics.get("rms_domega", 0.0),
                "rms_du": metrics.get("rms_du", 0.0),
                "saturation_hits": metrics.get("saturation_hits", 0.0),
                "overshoot_hits": metrics.get("overshoot_hits", 0.0),
            })

        # 新增：仅在 test_env_mode=True 时写 CSV
        self._write_step_csv(info)

        # 渲染数据打包
        # self.render_data = self._pack_render_data(obs, self.step_action, self.step_reward)

        self.step_action = np.asarray(action, dtype=np.float32).ravel()
        self.step_reward = float(reward)
        self.render_data = self._pack_render_data(obs, self.step_action, self.step_reward)

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
    def _get_observation(self,sDesCalc) -> np.ndarray:
        pos_ref = sDesCalc[0:3]
        vel_ref = sDesCalc[3:6]
        eul_ref = _quaternion_to_euler(sDesCalc[9:13])
        omega_ref = sDesCalc[13:16]

        pos_fb = self.quad.pos
        vel_fb = self.quad.vel
        eul_fb = self.quad.euler
        omega_fb = self.quad.omega

        obs = np.concatenate([
            pos_ref, vel_ref, eul_ref, omega_ref,
            pos_fb, vel_fb, eul_fb, omega_fb
        ])
        return obs.astype(np.float32)

        # 将obs,action,reward打包为一维数组，便于渲染    
    # def _pack_render_data(self, obs, action, reward):
    #     obs = np.ravel(obs)  # (n,)
    #     # 离散动作：转为标量数组；连续动作：拍扁
    #     if np.isscalar(action):
    #         action = np.array([action])
    #     else:
    #         action = np.ravel(action)
    #     reward = np.atleast_1d(reward).astype(np.float32)  # (1,)
    #     return np.concatenate([obs, action, reward], axis=0)
    
    def _pack_render_data(self, obs, action=None, reward=None):
        import numpy as np
        # 统一类型
        try:
            import torch
            if action is not None and isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            if reward is not None and isinstance(reward, torch.Tensor):
                reward = reward.detach().cpu().item()
        except Exception:
            pass

        act1d = None if action is None else np.asarray(action, dtype=np.float32).ravel()
        rew   = None if reward is None else float(np.asarray(reward).reshape(()))

        return {
            "obs": np.asarray(obs),   # 原样保留
            "action": act1d,          # 形状 (20,)
            "reward": rew,            # 标量
        }


    # =========================
    # 奖励设计
    # =========================

    # 统一的指标/误差提取（供奖励与窗口累计复用）
    def _extract_metrics(self, sDesCalc: np.ndarray) -> Dict[str, Any]:
        """
        从 sDesCalc 及当前状态提取误差指标。
        返回:
            - pos_err_xy, pos_err_z
            - vel_err_xy, vel_err_z
            - att_norm, att_err_roll, att_err_pitch, att_err_yaw
            - omega_norm, omega_err_p/q/r
            - pos_err_vec, vel_err_vec
        """
        # 参考状态
        pos_ref = sDesCalc[0:3]
        vel_ref = sDesCalc[3:6]
        eul_ref = _quaternion_to_euler(sDesCalc[9:13])
        omega_ref = sDesCalc[13:16]

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
        return float(self.dense_reward_scale * dense)

    def _collect_step_sample(self, m: Dict[str, Any], prev_cache: Dict[str, Any]) -> Dict[str, float]:
        """
        在本步子步中收集一次原子指标，用于步末聚合。
        prev_cache: 用于计算 dv/domega/du 的上一子步缓存（本步内局部）。
        返回当前子步的样本字典。
        """
        # 控制能量
        if hasattr(self.ctrl, 'w_cmd') and self.ctrl.w_cmd is not None:
            control_effort = float(np.sum(np.square(self.ctrl.w_cmd)))
        else:
            control_effort = 0.0

        # 平滑性（相邻子步差分，本步内）
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

        # 归一到每秒变化率（与原逻辑一致）
        if self.limits.get("normalize_rate_by_dt", True) and self.dt > 0:
            inv_dt = 1.0 / self.dt
            dv *= inv_dt
            domega *= inv_dt
            du *= inv_dt

        # 更新缓存
        prev_cache["vel"] = np.copy(self.quad.vel)
        prev_cache["omega"] = np.copy(self.quad.omega)
        if hasattr(self.ctrl, "w_cmd") and self.ctrl.w_cmd is not None:
            prev_cache["w_cmd"] = np.copy(self.ctrl.w_cmd)

        # 饱和判定
        saturation_hit = 0
        if hasattr(self.ctrl, 'w_cmd') and self.ctrl.w_cmd is not None:
            wmin, wmax = self.limits["w_cmd_min"], self.limits["w_cmd_max"]
            if np.any(self.ctrl.w_cmd < wmin) or np.any(self.ctrl.w_cmd > wmax):
                saturation_hit = 1

        # z 方向超调判定
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
            "u2": control_effort,
            "dv2": dv**2,
            "domega2": domega**2,
            "du2": du**2,
            "sat_hit": saturation_hit,
            "over_hit": overshoot_hit,
        }
        return entry

    def _compute_segment_reward_step(self, samples: List[Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
        """
        基于“本步子步样本”计算一次分段奖励。
        与原来的滑窗不同，这里对本步样本做简单均值/RMSE，不再做时间权重。
        """
        if len(samples) == 0:
            return 0.0, {}

        import numpy as np

        # 将样本堆叠
        keys = ["pos_xy2", "pos_z2", "vel_xy2", "vel_z2",
                "att_roll2", "att_pitch2", "omega2",
                "u2", "dv2", "domega2", "du2",
                "sat_hit", "over_hit"]
        S = {k: np.array([e[k] for e in samples], dtype=float) for k in keys}

        # RMSE或均值
        rmse_pos_xy   = float(np.sqrt(np.mean(S["pos_xy2"])))
        rmse_pos_z    = float(np.sqrt(np.mean(S["pos_z2"])))
        rmse_vel_xy   = float(np.sqrt(np.mean(S["vel_xy2"])))
        rmse_vel_z    = float(np.sqrt(np.mean(S["vel_z2"])))
        rmse_att_roll = float(np.sqrt(np.mean(S["att_roll2"])))
        rmse_att_pitch= float(np.sqrt(np.mean(S["att_pitch2"])))
        rms_omega     = float(np.sqrt(np.mean(S["omega2"])))

        mean_control_effort = float(np.mean(S["u2"]))
        rms_dv      = float(np.sqrt(np.mean(S["dv2"])))
        rms_domega  = float(np.sqrt(np.mean(S["domega2"])))
        rms_du      = float(np.sqrt(np.mean(S["du2"])))

        # 计数：总和更接近“次数”
        saturation_hits = float(np.sum(S["sat_hit"]))
        saturation_hits_norm = saturation_hits / 100    # 归一化，避免过大影响
        overshoot_hits  = float(np.sum(S["over_hit"]))
        overshoot_hits_norm = overshoot_hits / 100      # 归一化，避免过大影响

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
            sw["W_sat"]    * saturation_hits_norm +
            sw["W_over"]   * overshoot_hits_norm    
        )

        metrics = {
            'segment_duration': float(len(samples) * self.dt),
            'rmse_pos_xy': rmse_pos_xy,
            'rmse_pos_z': rmse_pos_z,
            'rmse_vel_xy': rmse_vel_xy,
            'rmse_vel_z': rmse_vel_z,
            'rmse_att_roll': rmse_att_roll,
            'rmse_att_pitch': rmse_att_pitch,
            'rms_omega': rms_omega,
            'mean_control_effort': mean_control_effort,
            'rms_dv': rms_dv,
            'rms_domega': rms_domega,
            'rms_du': rms_du,
            'saturation_hits': saturation_hits,
            'overshoot_hits': overshoot_hits,
            'segment_reward': float(segment_reward),
            'samples': int(len(samples)),
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
            'position_error': pos_error,
        }
        if self.last_segment_metrics is not None:
            info['last_segment'] = self.last_segment_metrics
            info['segment_time_accum'] = self.last_segment_metrics.get('segment_duration', 0.0)
        else:
            info['segment_time_accum'] = 0.0
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

    # =========================
    # 渲染与关闭
    # =========================

    # self.render_data = [obs, action, reward]
    # def _update_figure_from_render_data(self):
    #     rd = getattr(self, "render_data", None)
    #     if rd is None:
    #         return False

    #     # 解析 rd: 8 段各 3 维，共 24 长度
    #     try:
    #         pos_ref = rd[0:3];   vel_ref = rd[3:6];   eul_ref = rd[6:9];   omg_ref = rd[9:12]
    #         pos_fb  = rd[12:15]; vel_fb  = rd[15:18]; eul_fb  = rd[18:21]; omg_fb  = rd[21:24]
    #     except Exception:
    #         return False

    #     # 确保已初始化 figure/axes/lines
    #     if not getattr(self, "_render_inited", False):
    #         import matplotlib.pyplot as plt
    #         plt.ion() if self.render_mode == "human" else plt.ioff()
    #         self._fig, self._axs = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)
    #         self._lines = {"pos": {"ref": [], "fb": []},
    #                     "vel": {"ref": [], "fb": []},
    #                     "eul": {"ref": [], "fb": []},
    #                     "omg": {"ref": [], "fb": []}}
    #         self._t_hist = []
    #         self._pos_ref_hist, self._pos_fb_hist = [], []
    #         self._vel_ref_hist, self._vel_fb_hist = [], []
    #         self._eul_ref_hist, self._eul_fb_hist = [], []
    #         self._omg_ref_hist, self._omg_fb_hist = [], []

    #         self._axs[0].set_title("Position (x,y,z)");      self._axs[0].set_ylabel("pos [m]")
    #         self._axs[1].set_title("Velocity (vx,vy,vz)");   self._axs[1].set_ylabel("vel [m/s]")
    #         self._axs[2].set_title("Euler (roll,pitch,yaw) [rad]"); self._axs[2].set_ylabel("angle [rad]")
    #         self._axs[3].set_title("Body rates (wx,wy,wz) [rad/s]"); self._axs[3].set_ylabel("rate [rad/s]")
    #         self._axs[3].set_xlabel("Step")
    #         colors = ["tab:blue", "tab:orange", "tab:green"]
    #         for k, ax in zip(["pos", "vel", "eul", "omg"], self._axs):
    #             for i in range(3):
    #                 (lref,) = ax.plot([], [], linestyle="--", color=colors[i], alpha=0.8, label=f"{k}[{i}]_ref")
    #                 (lfb,)  = ax.plot([], [], linestyle="-",  color=colors[i], alpha=0.9, label=f"{k}[{i}]_fb")
    #                 self._lines[k]["ref"].append(lref)
    #                 self._lines[k]["fb"].append(lfb)
    #             ax.grid(True, alpha=0.3)
    #             ax.legend(ncols=3, fontsize=8, loc="upper right")
    #         self._render_inited = True

    #     # 追加历史
    #     step_idx = self._t_hist[-1] + 1 if self._t_hist else 0
    #     self._t_hist.append(step_idx)
    #     self._pos_ref_hist.append(pos_ref); self._pos_fb_hist.append(pos_fb)
    #     self._vel_ref_hist.append(vel_ref); self._vel_fb_hist.append(vel_fb)
    #     self._eul_ref_hist.append(eul_ref); self._eul_fb_hist.append(eul_fb)
    #     self._omg_ref_hist.append(omg_ref); self._omg_fb_hist.append(omg_fb)

    #     # 数组化
    #     def to_arr(lst):
    #         arr = np.asarray(lst)
    #         if arr.ndim == 1:
    #             arr = arr.reshape(1, -1)
    #         return arr
    #     t = np.asarray(self._t_hist)
    #     pos_ref_arr = to_arr(self._pos_ref_hist); pos_fb_arr = to_arr(self._pos_fb_hist)
    #     vel_ref_arr = to_arr(self._vel_ref_hist); vel_fb_arr = to_arr(self._vel_fb_hist)
    #     eul_ref_arr = to_arr(self._eul_ref_hist); eul_fb_arr = to_arr(self._eul_fb_hist)
    #     omg_ref_arr = to_arr(self._omg_ref_hist); omg_fb_arr = to_arr(self._omg_fb_hist)

    #     # 更新线条
    #     for i in range(3):
    #         self._lines["pos"]["ref"][i].set_data(t, pos_ref_arr[:, i])
    #         self._lines["pos"]["fb"][i].set_data(t,  pos_fb_arr[:, i])
    #         self._lines["vel"]["ref"][i].set_data(t, vel_ref_arr[:, i])
    #         self._lines["vel"]["fb"][i].set_data(t,  vel_fb_arr[:, i])
    #         self._lines["eul"]["ref"][i].set_data(t, eul_ref_arr[:, i])
    #         self._lines["eul"]["fb"][i].set_data(t,  eul_fb_arr[:, i])
    #         self._lines["omg"]["ref"][i].set_data(t, omg_ref_arr[:, i])
    #         self._lines["omg"]["fb"][i].set_data(t,  omg_fb_arr[:, i])

    #     # 重新缩放
    #     for ax in self._axs:
    #         ax.relim()
    #         ax.autoscale_view(True, True, True)

    #     return True

    def _update_figure_from_render_data(self):
        rd = getattr(self, "render_data", None)
        if rd is None:
            return False

        # ---------- 1) 解析 render_data ----------
        if isinstance(rd, dict):
            obs = rd.get("obs", None)
            action = rd.get("action", None)
            reward = rd.get("reward", None)
        elif isinstance(rd, (list, tuple)):
            obs = rd[0] if len(rd) > 0 else None
            action = rd[1] if len(rd) > 1 else None
            reward = rd[2] if len(rd) > 2 else None
        else:
            obs, action, reward = rd, None, None 

        def as1d(x):
            if x is None:
                return None
            arr = np.asarray(x, dtype=np.float32)
            return arr.ravel()

        obs1d = as1d(obs)
        act1d = as1d(action)
        rew_val = None if reward is None else float(np.asarray(reward).reshape(()))

        if not getattr(self, "_debug_once", False):
            print("[render init] act1d:",
                None if act1d is None else (type(act1d), getattr(act1d, "shape", None), act1d[:5]),
                "rew_val:", rew_val)
            print("[render init] rd type:", type(getattr(self, "render_data", None)))
            self._debug_once = True

        # ---------- 2) 从 obs 中取出 8 组 3D ----------
        def take3(x):
            x = as1d(x)
            if x is None or x.size < 3:
                return None
            return x[:3]

        pos_ref = vel_ref = eul_ref = omg_ref = None
        pos_fb  = vel_fb  = eul_fb  = omg_fb  = None

        if isinstance(obs, dict):
            pos_ref = take3(obs.get("pos_ref"))
            vel_ref = take3(obs.get("vel_ref"))
            eul_ref = take3(obs.get("eul_ref"))
            omg_ref = take3(obs.get("omg_ref"))
            pos_fb  = take3(obs.get("pos_fb"))
            vel_fb  = take3(obs.get("vel_fb"))
            eul_fb  = take3(obs.get("eul_fb"))
            omg_fb  = take3(obs.get("omg_fb"))
        elif obs1d is not None and obs1d.size >= 24:
            pos_ref = obs1d[0:3];   vel_ref = obs1d[3:6];   eul_ref = obs1d[6:9];   omg_ref = obs1d[9:12]
            pos_fb  = obs1d[12:15]; vel_fb  = obs1d[15:18]; eul_fb  = obs1d[18:21]; omg_fb  = obs1d[21:24]
        else:
            return False

        # 关键补丁：缺任何一组都直接跳过本帧
        required = (pos_ref, vel_ref, eul_ref, omg_ref, pos_fb, vel_fb, eul_fb, omg_fb)
        if any(x is None for x in required):
            return False

        need_act_axis = act1d is not None and act1d.size > 0
        need_rew_axis = rew_val is not None

        # ---------- 3) 初始化图像（两列布局） ----------
        if not getattr(self, "_render_inited", False):
            plt.ion() if getattr(self, "render_mode", None) == "human" else plt.ioff()

            from matplotlib.gridspec import GridSpec
            self._fig = plt.figure(figsize=(14, 9), constrained_layout=True)
            gs = GridSpec(nrows=3, ncols=2, figure=self._fig)

            # 两列共享同一个 x 轴（Step）
            ax_pos = self._fig.add_subplot(gs[0, 0])
            ax_vel = self._fig.add_subplot(gs[1, 0], sharex=ax_pos)
            ax_eul = self._fig.add_subplot(gs[2, 0], sharex=ax_pos)

            ax_omg = self._fig.add_subplot(gs[0, 1], sharex=ax_pos)
            ax_act = self._fig.add_subplot(gs[1, 1], sharex=ax_pos) if need_act_axis else None
            ax_rew = self._fig.add_subplot(gs[2, 1], sharex=ax_pos) if need_rew_axis else None

            # 记录轴对象（仅包含已创建的轴，便于后续统一 relim/autoscale）
            self._axs = [ax_pos, ax_vel, ax_eul, ax_omg]
            if ax_act is not None:
                self._axs.append(ax_act)
            if ax_rew is not None:
                self._axs.append(ax_rew)

            self._render_inited = True
            self._figure_shown = False  # for human

            # 线条容器
            self._lines = {
                "pos": {"ref": [], "fb": []},
                "vel": {"ref": [], "fb": []},
                "eul": {"ref": [], "fb": []},
                "omg": {"ref": [], "fb": []},
            }
            if need_act_axis:
                self._lines["act"] = {"all": []}
                self._act_dim = int(act1d.size)
            if need_rew_axis:
                self._lines["rew"] = {"all": []}

            # 历史缓存
            self._t_hist = []
            self._pos_ref_hist, self._pos_fb_hist = [], []
            self._vel_ref_hist, self._vel_fb_hist = [], []
            self._eul_ref_hist, self._eul_fb_hist = [], []
            self._omg_ref_hist, self._omg_fb_hist = [], []
            if need_act_axis:
                self._act_hist = []
            if need_rew_axis:
                self._rew_hist = []

            # 绘制左列：Position / Velocity / Euler
            colors = ["tab:blue", "tab:orange", "tab:green"]

            def add_group(ax, key, title, ylabel):
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                for i in range(3):
                    (lref,) = ax.plot([], [], "--", color=colors[i], alpha=0.85, label=f"{key}[{i}]_ref")
                    (lfb,)  = ax.plot([], [], "-",  color=colors[i], alpha=0.95, label=f"{key}[{i}]_fb")
                    self._lines[key]["ref"].append(lref)
                    self._lines[key]["fb"].append(lfb)
                ax.grid(True, alpha=0.3)
                ax.legend(ncols=3, fontsize=8, loc="upper right")

            add_group(ax_pos, "pos", "Position (x,y,z)", "pos [m]")
            add_group(ax_vel, "vel", "Velocity (vx,vy,vz)", "vel [m/s]")
            add_group(ax_eul, "eul", "Euler (roll,pitch,yaw) [rad]", "angle [rad]")

            # 绘制右列：Body rates / Action(可选) / Reward(可选)
            add_group(ax_omg, "omg", "Body rates (wx,wy,wz) [rad/s]", "rate [rad/s]")

            if ax_act is not None:
                ax_act.set_title("Action")
                ax_act.set_ylabel("act")
                act_colors = ["tab:purple", "tab:brown", "tab:cyan", "tab:pink", "tab:gray", "tab:olive"]
                for i in range(self._act_dim):
                    c = act_colors[i % len(act_colors)]
                    (l,) = ax_act.plot([], [], "-", color=c, alpha=0.95, label=f"act[{i}]")
                    self._lines["act"]["all"].append(l)
                ax_act.grid(True, alpha=0.3)
                ax_act.legend(ncols=min(6, self._act_dim), fontsize=8, loc="upper right")

            if ax_rew is not None:
                ax_rew.set_title("Reward")
                ax_rew.set_ylabel("r")
                ax_rew.set_xlabel("Step")
                (l,) = ax_rew.plot([], [], "-", color="tab:red", alpha=0.95, label="reward")
                self._lines["rew"]["all"].append(l)
                ax_rew.grid(True, alpha=0.3)
                ax_rew.legend(fontsize=8, loc="upper right")

            # 只在每列最底部显示 x 轴刻度
            for ax in [ax_pos, ax_vel, ax_omg, ax_act] if ax_act is not None else [ax_pos, ax_vel, ax_omg]:
                ax.label_outer()
            ax_eul.set_xlabel("Step")
            if ax_rew is None:
                # 如果没有 Reward，就让右列当前最底部的轴显示 X 轴标签
                (ax_act or ax_omg).set_xlabel("Step")

        # ---------- 4) 追加历史 ----------
        step_idx = self._t_hist[-1] + 1 if self._t_hist else 0
        self._t_hist.append(step_idx)
        self._pos_ref_hist.append(pos_ref); self._pos_fb_hist.append(pos_fb)
        self._vel_ref_hist.append(vel_ref); self._vel_fb_hist.append(vel_fb)
        self._eul_ref_hist.append(eul_ref); self._eul_fb_hist.append(eul_fb)
        self._omg_ref_hist.append(omg_ref); self._omg_fb_hist.append(omg_fb)
        if "act" in self._lines:
            if act1d is None:
                act1d = np.zeros(self._act_dim, dtype=np.float32)
            self._act_hist.append(act1d)
        if "rew" in self._lines:
            self._rew_hist.append(0.0 if rew_val is None else float(rew_val))

        # ---------- 5) 刷新线条 ----------
        def to_arr(lst):
            arr = np.asarray(lst, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr

        t = np.asarray(self._t_hist, dtype=np.int32)
        pos_ref_arr = to_arr(self._pos_ref_hist); pos_fb_arr = to_arr(self._pos_fb_hist)
        vel_ref_arr = to_arr(self._vel_ref_hist); vel_fb_arr = to_arr(self._vel_fb_hist)
        eul_ref_arr = to_arr(self._eul_ref_hist); eul_fb_arr = to_arr(self._eul_fb_hist)
        omg_ref_arr = to_arr(self._omg_ref_hist); omg_fb_arr = to_arr(self._omg_fb_hist)

        # 处理航向角，避免-pi到pi的跳变
        eul_ref_arr_unw = eul_ref_arr.copy()
        eul_fb_arr_unw  = eul_fb_arr.copy()
        eul_ref_arr_unw[:, 2] = np.unwrap(eul_ref_arr[:, 2])
        eul_fb_arr_unw[:, 2]  = np.unwrap(eul_fb_arr[:, 2])

        for i in range(3):
            self._lines["pos"]["ref"][i].set_data(t, pos_ref_arr[:, i])
            self._lines["pos"]["fb"][i].set_data(t,  pos_fb_arr[:, i])
            self._lines["vel"]["ref"][i].set_data(t, vel_ref_arr[:, i])
            self._lines["vel"]["fb"][i].set_data(t,  vel_fb_arr[:, i])
            self._lines["eul"]["ref"][i].set_data(t, eul_ref_arr_unw[:, i])
            self._lines["eul"]["fb"][i].set_data(t,  eul_fb_arr_unw[:, i])
            self._lines["omg"]["ref"][i].set_data(t, omg_ref_arr[:, i])
            self._lines["omg"]["fb"][i].set_data(t,  omg_fb_arr[:, i])

        if "act" in self._lines:
            act_arr = to_arr(self._act_hist)  # (T, A)
            for i, l in enumerate(self._lines["act"]["all"]):
                l.set_data(t, act_arr[:, i])

        if "rew" in self._lines:
            rew_arr = np.asarray(self._rew_hist, dtype=np.float32)  # (T,)
            self._lines["rew"]["all"][0].set_data(t, rew_arr)

        # ---------- 6) 重新缩放 ----------
        for ax in self._axs:
            ax.relim()
            ax.autoscale_view(True, True, True)

        return True
    
    # def render(self):
    #     if getattr(self, "render_mode", None) is None:
    #         return None

    #     if self.render_mode == "human":
    #         updated = self._update_figure_from_render_data()
    #         if updated:
    #             self._fig.canvas.draw()
    #             self._fig.canvas.flush_events()
    #         return None

    #     if self.render_mode == "rgb_array":
    #         # 先更新
    #         self._update_figure_from_render_data()

    #         # 截图
    #         canvas = FigureCanvasAgg(self._fig)
    #         canvas.draw()  # 关键：确保缓冲区是最新的
    #         buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    #         w, h = self._fig.canvas.get_width_height()
    #         img = buf.reshape((h, w, 4))[..., :3]

    #         # 保存
    #         save_frames = getattr(self, "save_frames", True)
    #         save_dir = getattr(self, "save_dir", "./log_render_frames")
    #         save_overwrite = getattr(self, "save_overwrite", True)
    #         # dpi = getattr(self, "save_dpi", None)  # 不用于 imageio，保留给 savefig 用
    #         if save_frames:
    #             os.makedirs(save_dir, exist_ok=True)
    #             if save_overwrite:
    #                 out_path = os.path.join(save_dir, "frame.png")
    #             else:
    #                 idx = getattr(self, "_frame_idx", 0)
    #                 out_path = os.path.join(save_dir, f"frame_{idx:06d}.png")
    #                 self._frame_idx = idx + 1
    #             imageio.imwrite(out_path, img)

    #         return img

    def render(self):
        mode = getattr(self, "render_mode", None)
        if mode is None:
            return None

        updated = self._update_figure_from_render_data()
        if not updated:
            # 可选：调试日志，帮助定位为何 update 失败
            # print("[render] update=False (render_data incomplete or invalid)")
            return None

        if mode == "human":
            # 首次显示窗口
            if not getattr(self, "_figure_shown", False):
                try:
                    plt.show(block=False)
                except Exception:
                    pass
                self._figure_shown = True

            self._fig.canvas.draw()
            try:
                self._fig.canvas.flush_events()
                plt.pause(0.001)  # 让 GUI 事件循环跑一下
            except Exception:
                pass
            return None

        if mode == "rgb_array":
            canvas = FigureCanvasAgg(self._fig)
            canvas.draw()
            buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            w, h = canvas.get_width_height()  # 用 canvas 的尺寸
            img = buf.reshape((h, w, 4))[..., :3]

            # 保存（可选）
            save_frames = getattr(self, "save_frames", True)
            save_dir = getattr(self, "save_dir", "./log_render_frames")
            save_overwrite = getattr(self, "save_overwrite", True)
            if save_frames:
                os.makedirs(save_dir, exist_ok=True)
                if save_overwrite:
                    out_path = os.path.join(save_dir, "frame.png")
                else:
                    idx = getattr(self, "_frame_idx", 0)
                    out_path = os.path.join(save_dir, f"frame_{idx:06d}.png")
                    self._frame_idx = idx + 1
                imageio.imwrite(out_path, img)
                # 可选：打印路径，便于确认
                # print(f"[render] saved {out_path}, img={img.shape} {img.dtype}")

            return img

        # 未知模式
        return None

    # 暂时没有用，需要测试结束后绘制全部的数据
    def plot(self):
        """
        mode: 绘图 + 动画（plt.show）
        """        
        try:
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

        except Exception as e:
            print(f"Render error: {e}")
            pass

    def _save_render_figure(self, path=None, dpi=150):
        if getattr(self, "_render_inited", False):
            import os
            os.makedirs(getattr(self, "save_dir", "./log_render_exports"), exist_ok=True)
            if path is None:
                ep = getattr(self, "_episode_index", 0)
                path = os.path.join(self.save_dir, f"episode_{ep:04d}.png")
            self._fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"[render] figure saved to: {path}")

    def close(self):
        try:
            self._save_render_figure()
        except Exception as e:
            print(f"[render] save on close failed: {e}")
        if getattr(self, "_render_inited", False):
            plt.close(self._fig)
            self._render_inited = False