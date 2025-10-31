
# -*- coding: utf-8 -*-
"""
四旋翼轨迹生成器 - 重构版
支持速度跟踪和位置跟踪的强化学习训练轨迹

作者: John Bass (原始版本)
重构: 添加了完整的RL训练轨迹库
更新: 拆分速度轨迹100为1001/1002/1003，拆分101为1011/1012
"""

import numpy as np
from numpy import pi
from numpy.linalg import norm
from pathlib import Path
from waypoints import makeWaypoints
import config

# ============================================================================
# 工具函数
# ============================================================================

def get_poly_cc(n, k, t):
    """获取 n 阶多项式在 t 时刻的 k 阶导数的系数。
    
    Args:
        n: 多项式阶数
        k: 导数阶数
        t: 时间点
    
    Returns:
        系数数组
    """
    assert (n > 0 and k >= 0), "order and derivative must be positive."
    cc = np.ones(n)
    D = np.linspace(n-1, 0, n)
    for i in range(n):
        for _ in range(k):
            cc[i] = cc[i] * D[i]
            D[i] = D[i] - 1
            if D[i] == -1:
                D[i] = 0
    for i, c in enumerate(cc):
        cc[i] = c * np.power(t, D[i])
    return cc


# ============================================================================
# 主轨迹类
# ============================================================================

class Trajectory:
    """四旋翼轨迹生成器
    
    控制类型 (ctrlType):
        - "xyz_vel": 速度控制模式
        - "xyz_pos": 位置控制模式
        - "xy_vel_z_pos": 混合模式（XY速度 + Z位置）
    
    轨迹类型编码 (xyzType):
        速度跟踪轨迹 (100-199):
            *** 新拆分轨迹 ***
            1001: 俯仰轴阶跃测试 (30秒)
            1002: 滚转轴阶跃测试 (30秒)
            * 1003: 俯仰-滚转耦合阶跃测试 (30秒)
            1011: 正弦扫频扫幅 - 低频段 (30秒)
            * 1012: 正弦扫频扫幅 - 高频段 (30秒)
            
            *** 其他轨迹 ***
            102: 方波速度
            103: 三角波速度
            104: 随机组合速度
            105: 3D李萨如速度
        
        位置跟踪轨迹 (200-299):
            200: 定高随机阶跃
            * 201: 典型轨迹组合（8字+螺旋+圆形）
            202: 圆形轨迹
            * 203: 8字轨迹
            204: 方形轨迹
            205: 螺旋上升
            206: 3D李萨如
            207: 随机多项式轨迹
    
    航向类型 (yawType):
        0: 固定航向（0度）
        1: 按时间设置航向
        2: 航向插值
        3: 航向跟随速度方向（推荐）
        4: 固定为0
    """

    def __init__(self, quad, ctrlType, trajSelect):
        """初始化轨迹生成器
        
        Args:
            quad: 四旋翼对象
            ctrlType: 控制类型字符串
            trajSelect: [xyzType, yawType, averVel]
        """
        # ========== 基本配置 ==========
        self.ctrlType = ctrlType
        self.xyzType = trajSelect[0]
        self.yawType = trajSelect[1]
        self.averVel = trajSelect[2]
        
        # ========== 航点数据（用于传统轨迹） ==========
        t_wps, wps, y_wps, v_wp = makeWaypoints()
        self.t_wps = t_wps
        self.wps = wps
        self.y_wps = y_wps
        self.v_wp = v_wp
        
        # ========== 状态变量 ==========
        self.end_reached = 0
        self.t_idx = 0
        self.t_arrived = 0
        self.arrived = False
        self.prevDesYaw = 0.0
        self.current_heading = quad.psi
        
        # ========== 期望状态向量 ==========
        self.desPos = np.zeros(3)
        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)
        self.desThr = np.zeros(3)
        self.desEul = np.zeros(3)
        self.desPQR = np.zeros(3)
        self.desYawRate = 0.
        
        # ========== 根据控制类型初始化 ==========
        if self.ctrlType in ["xyz_vel", "xy_vel_z_pos"]:
            self._init_velocity_trajectories()
        elif self.ctrlType == "xyz_pos":
            self._init_position_trajectories()
        
        # ========== 初始状态 ==========
        self.sDes = np.hstack((
            self.desPos, self.desVel, self.desAcc, self.desThr,
            self.desEul, self.desPQR, self.desYawRate
        )).astype(float)

    # ========================================================================
    # 初始化方法
    # ========================================================================
    
    def _init_velocity_trajectories(self):
        """初始化速度跟踪轨迹参数"""
        
        # ==================== 新拆分轨迹参数 ====================
        if self.xyzType in [1001, 1002, 1003]:
            # 俯仰/滚转/耦合阶跃测试（各30秒）
            self.vel_step_duration = 5.0
            self.vel_zero_duration = 5.0
            self.vel_cycle_time = 10.0
            # self.vel_levels = [2.0, 5.0, 10.0]
            self.vel_levels = [5.0, 5.0, 5.0]
            self.num_steps = len(self.vel_levels)
            self.roll_pitch_ratio = 0.6
            self.current_step = 0
            self.last_cycle_start_time = 0
            self.current_vel_target = np.zeros(3)
        
        elif self.xyzType in [1011, 1012]:
            # 正弦扫频扫幅 - 拆分版（各30秒）
            self.vel_amplitude_min = 3.0
            self.vel_amplitude_max = 10.0
            self.vel_amplitude_period = 30.0
            self.vel_axis_ratio = np.array([0.6, 0.5, 0.3])
            self.vel_phase = np.array([0, pi/3, 2*pi/3])
            
            if self.xyzType == 1011:
                # 低频段：0.01 - 0.055 Hz
                self.vel_freq_min = 0.01
                self.vel_freq_max = 0.055
                self.vel_freq_period = 30.0
            else:  # 1012
                # 高频段：0.055 - 0.1 Hz
                self.vel_freq_min = 0.055
                self.vel_freq_max = 0.1
                self.vel_freq_period = 30.0
        
        # ==================== 原有轨迹参数（保留兼容性） ====================
        elif self.xyzType == 100:
            # 原完整版（90秒）
            self.vel_step_duration = 5.0
            self.vel_zero_duration = 5.0
            self.vel_cycle_time = 10.0
            self.vel_levels = [2.0, 5.0, 10.0]
            self.num_steps = len(self.vel_levels)
            self.roll_pitch_ratio = 0.6
            self.current_step = 0
            self.last_cycle_start_time = 0
            self.current_vel_target = np.zeros(3)
            self.phase1_end = self.num_steps * self.vel_cycle_time
            self.phase2_end = 2 * self.num_steps * self.vel_cycle_time
            self.phase3_end = 3 * self.num_steps * self.vel_cycle_time
        
        elif self.xyzType == 101:
            # 原完整版（60秒）
            self.vel_amplitude_min = 3.0
            self.vel_amplitude_max = 10.0
            self.vel_amplitude_period = 30.0
            self.vel_freq_min = 0.01
            self.vel_freq_max = 0.1
            self.vel_freq_period = 60.0
            self.vel_axis_ratio = np.array([0.6, 0.5, 0.3])
            self.vel_phase = np.array([0, pi/3, 2*pi/3])
        
        # ==================== 其他轨迹 ====================
        elif self.xyzType == 102:
            # 方波速度
            self.square_wave_period = 10.0
            self.square_wave_amplitude = 5.0
            self.square_wave_axes = [True, True, False]
        
        elif self.xyzType == 103:
            # 三角波速度
            self.triangle_wave_period = 8.0
            self.triangle_wave_amplitude = 6.0
            self.triangle_wave_axes = [True, True, True]
        
        elif self.xyzType == 104:
            # 随机组合速度
            np.random.seed(42)
            self.random_vel_duration = 5.0
            self.random_vel_max = 8.0
            self.last_random_time = 0
            self.current_random_vel = np.zeros(3)
        
        elif self.xyzType == 105:
            # 3D李萨如速度曲线
            self.lissajous_vel_amp = np.array([5.0, 4.0, 2.0])
            self.lissajous_vel_freq = np.array([0.5, 0.7, 0.3])
            self.lissajous_vel_phase = np.array([0, pi/4, pi/2])

    def _init_position_trajectories(self):
        """初始化位置跟踪轨迹参数"""
        
        # 计算航段时间
        self.T_segment = np.diff(self.t_wps)
        if self.averVel == 1:
            distance_segment = self.wps[1:] - self.wps[:-1]
            self.T_segment = np.sqrt(
                distance_segment[:,0]**2 + 
                distance_segment[:,1]**2 + 
                distance_segment[:,2]**2
            ) / self.v_wp
            self.t_wps = np.zeros(len(self.T_segment) + 1)
            self.t_wps[1:] = np.cumsum(self.T_segment)
        
        if self.xyzType == 200:
            # 定高随机阶跃
            self.pos_step_time = 5.0
            self.pos_height = 2.0
            self.pos_step_min = 2.0
            self.pos_step_max = 10.0
            self.last_pos_step_time = 0
            self.current_pos_target = np.array([0, 0, self.pos_height])
            np.random.seed(123)
        
        elif self.xyzType == 201:
            # 典型轨迹组合
            self.phase_duration = 15.0
        
        elif self.xyzType == 202:
            # 圆形轨迹
            self.circle_radius = 8.0
            self.circle_omega = 0.3
            self.circle_height = 2.0
            self.circle_center = np.array([0., 0.])
        
        elif self.xyzType == 203:
            # 8字轨迹
            self.eight_amp_x = 6.0
            self.eight_amp_y = 4.0
            self.eight_omega = 0.4
            self.eight_height = 2.5
        
        elif self.xyzType == 204:
            # 方形轨迹
            self.square_size = 10.0
            self.square_velocity = 2.0
            self.square_height = 2.0
            self.square_waypoints = np.array([
                [0, 0],
                [self.square_size, 0],
                [self.square_size, self.square_size],
                [0, self.square_size],
                [0, 0]
            ])
            self.square_total_length = 4 * self.square_size
        
        elif self.xyzType == 205:
            # 螺旋上升
            self.helix_radius = 5.0
            self.helix_omega = 0.5
            self.helix_climb_rate = 0.5
            self.helix_initial_height = 1.0
            self.helix_max_height = 5.0
        
        elif self.xyzType == 206:
            # 3D李萨如
            self.lissajous_amp = np.array([5.0, 4.0, 1.5])
            self.lissajous_freq = np.array([1.0, 2.0, 3.0])
            self.lissajous_phase = np.array([0, pi/3, pi/2])
            self.lissajous_center = np.array([0., 0., 2.5])
        
        elif self.xyzType == 207:
            # 随机多项式轨迹
            np.random.seed(456)
            self.poly_duration = 20.0
            self.poly_degree = 5
            self.poly_v_max = 5.0
            self.poly_a_max = 3.0
            self._generate_random_polynomial()
        
        elif self.xyzType == 208:
            # 航点巡航
            self.waypoint_list = np.array([
                [0, 0, 2],
                [10, 5, 2.5],
                [15, -3, 3],
                [8, -10, 2.5],
                [0, 0, 2]
            ])
            self.waypoint_velocity = 3.0

    # ========================================================================
    # 速度跟踪轨迹生成方法 (新拆分版本)
    # ========================================================================
    
    def _vel_pitch_only_step(self, t):
        """xyzType=1001: 俯仰轴阶跃测试（30秒）"""
        desVel = np.zeros(3)
        
        current_step = int(t // self.vel_cycle_time)
        if current_step >= self.num_steps:
            return desVel
        
        time_in_cycle = t - current_step * self.vel_cycle_time
        current_vel = self.vel_levels[current_step]
        is_step_phase = (time_in_cycle < self.vel_step_duration)
        
        # 仅俯仰轴（X轴）
        desVel[0] = current_vel if is_step_phase else 0.0
        
        return desVel
    
    def _vel_roll_only_step(self, t):
        """xyzType=1002: 滚转轴阶跃测试（30秒）"""
        desVel = np.zeros(3)
        
        current_step = int(t // self.vel_cycle_time)
        if current_step >= self.num_steps:
            return desVel
        
        time_in_cycle = t - current_step * self.vel_cycle_time
        current_vel = self.vel_levels[current_step]
        is_step_phase = (time_in_cycle < self.vel_step_duration)
        
        # 仅滚转轴（Y轴）
        desVel[1] = current_vel if is_step_phase else 0.0
        
        return desVel
    
    def _vel_coupled_step(self, t):
        """xyzType=1003: 俯仰-滚转耦合阶跃测试（30秒）"""
        desVel = np.zeros(3)
        
        current_step = int(t // self.vel_cycle_time)
        if current_step >= self.num_steps:
            return desVel
        
        time_in_cycle = t - current_step * self.vel_cycle_time
        current_vel = self.vel_levels[current_step]
        is_step_phase = (time_in_cycle < self.vel_step_duration)
        
        # 俯仰-滚转耦合
        if is_step_phase:
            desVel[0] = current_vel
            desVel[1] = current_vel * self.roll_pitch_ratio
        
        return desVel
    
    def _vel_sine_sweep_low_freq(self, t):
        """xyzType=1011: 正弦扫频扫幅 - 低频段（30秒）"""
        return self._vel_sine_sweep_generic(t)
    
    def _vel_sine_sweep_high_freq(self, t):
        """xyzType=1012: 正弦扫频扫幅 - 高频段（30秒）"""
        return self._vel_sine_sweep_generic(t)
    
    def _vel_sine_sweep_generic(self, t):
        """正弦扫频扫幅通用实现"""
        desVel = np.zeros(3)
        desAcc = np.zeros(3)
        
        # 幅值扫描（三角波）
        amp_ratio = np.abs((t % self.vel_amplitude_period) / self.vel_amplitude_period * 2 - 1)
        current_amp = self.vel_amplitude_min + (self.vel_amplitude_max - self.vel_amplitude_min) * amp_ratio
        
        # 频率扫描（线性）
        freq_ratio = (t % self.vel_freq_period) / self.vel_freq_period
        current_freq = self.vel_freq_min + (self.vel_freq_max - self.vel_freq_min) * freq_ratio
        omega = 2 * pi * current_freq
        
        # 归一化
        norm_factor = np.sqrt(np.sum(self.vel_axis_ratio**2))
        
        # 各轴分解
        for i in range(3):
            ratio_norm = self.vel_axis_ratio[i] / norm_factor
            desVel[i] = current_amp * ratio_norm * np.sin(omega * t + self.vel_phase[i])
            desAcc[i] = current_amp * ratio_norm * omega * np.cos(omega * t + self.vel_phase[i])
        
        return desVel, desAcc

    # ========================================================================
    # 速度跟踪轨迹生成方法 (原有版本 - 保留兼容性)
    # ========================================================================
    
    def _vel_pitch_roll_step(self, t):
        """xyzType=100: 俯仰-滚转阶跃测试（完整90秒）"""
        desVel = np.zeros(3)
        
        if t < self.phase1_end:
            phase_time = t
            phase_mode = 1
        elif t < self.phase2_end:
            phase_time = t - self.phase1_end
            phase_mode = 2
        elif t < self.phase3_end:
            phase_time = t - self.phase2_end
            phase_mode = 4
        else:
            return desVel
        
        current_step = int(phase_time // self.vel_cycle_time)
        if current_step >= self.num_steps:
            current_step = self.num_steps - 1
        
        time_in_cycle = phase_time - current_step * self.vel_cycle_time
        current_vel = self.vel_levels[current_step]
        is_step_phase = (time_in_cycle < self.vel_step_duration)
        
        if phase_mode == 1:  # 仅俯仰
            desVel[0] = current_vel if is_step_phase else 0.0
        elif phase_mode == 2:  # 仅滚转
            desVel[1] = current_vel if is_step_phase else 0.0
        elif phase_mode == 4:  # 耦合
            if is_step_phase:
                desVel[0] = current_vel
                desVel[1] = current_vel * self.roll_pitch_ratio
        
        return desVel
    
    def _vel_sine_sweep(self, t):
        """xyzType=101: 正弦扫频扫幅（完整60秒）"""
        return self._vel_sine_sweep_generic(t)
    
    def _vel_square_wave(self, t):
        """xyzType=102: 方波速度"""
        desVel = np.zeros(3)
        
        for i in range(3):
            if self.square_wave_axes[i]:
                phase = (t % self.square_wave_period) / self.square_wave_period
                desVel[i] = self.square_wave_amplitude if phase < 0.5 else -self.square_wave_amplitude
        
        return desVel
    
    def _vel_triangle_wave(self, t):
        """xyzType=103: 三角波速度"""
        desVel = np.zeros(3)
        desAcc = np.zeros(3)
        
        for i in range(3):
            if self.triangle_wave_axes[i]:
                phase = (t % self.triangle_wave_period) / self.triangle_wave_period
                if phase < 0.25:
                    desVel[i] = self.triangle_wave_amplitude * (4 * phase)
                    desAcc[i] = self.triangle_wave_amplitude * 4 / self.triangle_wave_period
                elif phase < 0.75:
                    desVel[i] = self.triangle_wave_amplitude * (2 - 4 * phase)
                    desAcc[i] = -self.triangle_wave_amplitude * 4 / self.triangle_wave_period
                else:
                    desVel[i] = self.triangle_wave_amplitude * (4 * phase - 4)
                    desAcc[i] = self.triangle_wave_amplitude * 4 / self.triangle_wave_period
        
        return desVel, desAcc
    
    def _vel_random_combined(self, t):
        """xyzType=104: 随机组合速度"""
        if t >= self.last_random_time + self.random_vel_duration:
            self.current_random_vel = np.random.uniform(
                -self.random_vel_max, 
                self.random_vel_max, 
                3
            )
            self.current_random_vel[2] *= 0.5
            self.last_random_time = t
        
        return self.current_random_vel
    
    def _vel_lissajous_3d(self, t):
        """xyzType=105: 3D李萨如速度曲线"""
        desVel = np.zeros(3)
        desAcc = np.zeros(3)
        
        for i in range(3):
            omega = 2 * pi * self.lissajous_vel_freq[i]
            desVel[i] = self.lissajous_vel_amp[i] * np.sin(omega * t + self.lissajous_vel_phase[i])
            desAcc[i] = self.lissajous_vel_amp[i] * omega * np.cos(omega * t + self.lissajous_vel_phase[i])
        
        return desVel, desAcc

    # ========================================================================
    # 位置跟踪轨迹生成方法 (200-299)
    # ========================================================================
    
    def _pos_random_step(self, t):
        """xyzType=200: 定高随机阶跃"""
        if t >= self.last_pos_step_time + self.pos_step_time:
            distance = 0.0
            direction = np.array([1.0, 0.0])
            while distance < self.pos_step_min or distance > self.pos_step_max:
                theta = np.random.uniform(0.0, 2.0 * pi)
                distance = np.random.uniform(self.pos_step_min, self.pos_step_max)
                direction = np.array([np.cos(theta), np.sin(theta)])
            
            new_xy = self.current_pos_target[:2] + distance * direction
            self.current_pos_target = np.array([new_xy[0], new_xy[1], self.pos_height])
            self.last_pos_step_time = t
        
        self.desPos = self.current_pos_target.copy()
        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)
    
    def _pos_typical_combined(self, t):
        """xyzType=201: 典型轨迹组合（8字+螺旋+圆形）"""
        total_cycle_time = 3 * self.phase_duration
        t_cycle = np.mod(t, total_cycle_time)
        
        if t_cycle < self.phase_duration:
            self._pos_lissajous_eight(t_cycle)
        elif t_cycle < 2 * self.phase_duration:
            self._pos_helix(t_cycle - self.phase_duration)
        else:
            self._pos_circle_internal(t_cycle - 2 * self.phase_duration)
    
    def _pos_lissajous_eight(self, t):
        """李萨如8字轨迹（内部方法）"""
        A, B = 18.0, 14.0
        freq_ratio = 2
        omega = 2 * pi / self.phase_duration
        height = 2.0
        
        self.desPos[0] = A * np.sin(omega * t)
        self.desPos[1] = B * np.sin(freq_ratio * omega * t)
        self.desPos[2] = height
        
        self.desVel[0] = A * omega * np.cos(omega * t)
        self.desVel[1] = B * freq_ratio * omega * np.cos(freq_ratio * omega * t)
        self.desVel[2] = 0
        
        self.desAcc[0] = -A * omega**2 * np.sin(omega * t)
        self.desAcc[1] = -B * (freq_ratio * omega)**2 * np.sin(freq_ratio * omega * t)
        self.desAcc[2] = 0
    
    def _pos_helix(self, t):
        """螺旋轨迹（内部方法）"""
        radius = 12.0
        omega = 4 * pi / self.phase_duration
        climb_rate = 6.0 / self.phase_duration
        initial_height = 2.0
        
        self.desPos[0] = radius * np.cos(omega * t)
        self.desPos[1] = radius * np.sin(omega * t)
        self.desPos[2] = initial_height + climb_rate * t
        
        self.desVel[0] = -radius * omega * np.sin(omega * t)
        self.desVel[1] = radius * omega * np.cos(omega * t)
        self.desVel[2] = climb_rate
        
        self.desAcc[0] = -radius * omega**2 * np.cos(omega * t)
        self.desAcc[1] = -radius * omega**2 * np.sin(omega * t)
        self.desAcc[2] = 0
    
    def _pos_circle_internal(self, t):
        """圆形轨迹（内部方法）"""
        radius = 15.0
        omega = 2 * pi / self.phase_duration
        height = 1.5
        
        self.desPos[0] = radius * np.cos(omega * t)
        self.desPos[1] = radius * np.sin(omega * t)
        self.desPos[2] = height
        
        self.desVel[0] = -radius * omega * np.sin(omega * t)
        self.desVel[1] = radius * omega * np.cos(omega * t)
        self.desVel[2] = 0
        
        self.desAcc[0] = -radius * omega**2 * np.cos(omega * t)
        self.desAcc[1] = -radius * omega**2 * np.sin(omega * t)
        self.desAcc[2] = 0
    
    def _pos_circle(self, t):
        """xyzType=202: 独立圆形轨迹"""
        omega = self.circle_omega
        
        self.desPos[0] = self.circle_center[0] + self.circle_radius * np.cos(omega * t)
        self.desPos[1] = self.circle_center[1] + self.circle_radius * np.sin(omega * t)
        self.desPos[2] = self.circle_height
        
        self.desVel[0] = -self.circle_radius * omega * np.sin(omega * t)
        self.desVel[1] = self.circle_radius * omega * np.cos(omega * t)
        self.desVel[2] = 0
        
        self.desAcc[0] = -self.circle_radius * omega**2 * np.cos(omega * t)
        self.desAcc[1] = -self.circle_radius * omega**2 * np.sin(omega * t)
        self.desAcc[2] = 0
    
    def _pos_figure_eight(self, t):
        """xyzType=203: 8字轨迹"""
        omega = self.eight_omega
        
        self.desPos[0] = self.eight_amp_x * np.sin(omega * t)
        self.desPos[1] = self.eight_amp_y * np.sin(2 * omega * t)
        self.desPos[2] = self.eight_height
        
        self.desVel[0] = self.eight_amp_x * omega * np.cos(omega * t)
        self.desVel[1] = self.eight_amp_y * 2 * omega * np.cos(2 * omega * t)
        self.desVel[2] = 0
        
        self.desAcc[0] = -self.eight_amp_x * omega**2 * np.sin(omega * t)
        self.desAcc[1] = -self.eight_amp_y * 4 * omega**2 * np.sin(2 * omega * t)
        self.desAcc[2] = 0
    
    def _pos_square(self, t):
        """xyzType=204: 方形轨迹"""
        perimeter = self.square_total_length
        period = perimeter / self.square_velocity
        t_mod = np.mod(t, period)
        distance = self.square_velocity * t_mod
        
        side_length = self.square_size
        if distance < side_length:
            x = distance
            y = 0
            vx, vy = self.square_velocity, 0
        elif distance < 2 * side_length:
            x = side_length
            y = distance - side_length
            vx, vy = 0, self.square_velocity
        elif distance < 3 * side_length:
            x = side_length - (distance - 2 * side_length)
            y = side_length
            vx, vy = -self.square_velocity, 0
        else:
            x = 0
            y = side_length - (distance - 3 * side_length)
            vx, vy = 0, -self.square_velocity
        
        self.desPos = np.array([x, y, self.square_height])
        self.desVel = np.array([vx, vy, 0.0])
        self.desAcc = np.zeros(3)
    
    def _pos_spiral(self, t):
        """xyzType=205: 螺旋上升"""
        omega = self.helix_omega
        z = self.helix_initial_height + self.helix_climb_rate * t
        
        if z > self.helix_max_height:
            z = self.helix_max_height
            vz = 0
        else:
            vz = self.helix_climb_rate
        
        self.desPos[0] = self.helix_radius * np.cos(omega * t)
        self.desPos[1] = self.helix_radius * np.sin(omega * t)
        self.desPos[2] = z
        
        self.desVel[0] = -self.helix_radius * omega * np.sin(omega * t)
        self.desVel[1] = self.helix_radius * omega * np.cos(omega * t)
        self.desVel[2] = vz
        
        self.desAcc[0] = -self.helix_radius * omega**2 * np.cos(omega * t)
        self.desAcc[1] = -self.helix_radius * omega**2 * np.sin(omega * t)
        self.desAcc[2] = 0
    
    def _pos_lissajous_3d(self, t):
        """xyzType=206: 3D李萨如"""
        for i in range(3):
            omega = 2 * pi * self.lissajous_freq[i] / 20.0
            self.desPos[i] = self.lissajous_center[i] + \
                            self.lissajous_amp[i] * np.sin(omega * t + self.lissajous_phase[i])
            self.desVel[i] = self.lissajous_amp[i] * omega * \
                            np.cos(omega * t + self.lissajous_phase[i])
            self.desAcc[i] = -self.lissajous_amp[i] * omega**2 * \
                            np.sin(omega * t + self.lissajous_phase[i])
    
    def _generate_random_polynomial(self):
        """生成随机多项式轨迹系数"""
        self.poly_coeff = np.random.randn(3, self.poly_degree + 1) * 0.5
        self.poly_coeff *= 0.5
    
    def _pos_random_polynomial(self, t):
        """xyzType=207: 随机多项式轨迹"""
        t_norm = np.mod(t, self.poly_duration) / self.poly_duration
        
        for i in range(3):
            self.desPos[i] = np.polyval(self.poly_coeff[i], t_norm)
            vel_coeff = np.polyder(self.poly_coeff[i])
            self.desVel[i] = np.polyval(vel_coeff, t_norm) / self.poly_duration
            acc_coeff = np.polyder(vel_coeff)
            self.desAcc[i] = np.polyval(acc_coeff, t_norm) / (self.poly_duration**2)

    # ========================================================================
    # 航向控制方法
    # ========================================================================
    
    def _yaw_follow_velocity(self, t, Ts, quad):
        """航向跟随速度方向"""
        if np.linalg.norm(self.desVel[:2]) > 0.1:
            self.desEul[2] = np.arctan2(self.desVel[1], self.desVel[0])
        
        if (np.sign(self.desEul[2]) != np.sign(self.current_heading) and 
            abs(self.desEul[2] - self.current_heading) >= 2*pi - 0.1):
            self.current_heading += np.sign(self.desEul[2]) * 2*pi
        
        delta_psi = self.desEul[2] - self.current_heading
        self.desYawRate = delta_psi / Ts
        self.current_heading = self.desEul[2]

    # ========================================================================
    # 主接口：desiredState
    # ========================================================================
    
    def desiredState(self, t, Ts, quad):
        """计算期望状态
        
        Args:
            t: 当前时间
            Ts: 采样时间
            quad: 四旋翼对象
        
        Returns:
            sDes: 期望状态向量 [pos(3), vel(3), acc(3), thr(3), eul(3), pqr(3), yawRate(1)]
        """
        # 重置期望状态
        self.desPos = np.zeros(3)
        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)
        self.desThr = np.zeros(3)
        self.desEul = np.zeros(3)
        self.desPQR = np.zeros(3)
        self.desYawRate = 0.
        
        # ========== 速度控制模式 ==========
        if self.ctrlType in ["xyz_vel", "xy_vel_z_pos"]:
            # 新拆分轨迹
            if self.xyzType == 1001:
                self.desVel = self._vel_pitch_only_step(t)
            elif self.xyzType == 1002:
                self.desVel = self._vel_roll_only_step(t)
            elif self.xyzType == 1003:
                self.desVel = self._vel_coupled_step(t)
            elif self.xyzType == 1011:
                self.desVel, self.desAcc = self._vel_sine_sweep_low_freq(t)
            elif self.xyzType == 1012:
                self.desVel, self.desAcc = self._vel_sine_sweep_high_freq(t)
            # 原有轨迹
            elif self.xyzType == 100:
                self.desVel = self._vel_pitch_roll_step(t)
            elif self.xyzType == 101:
                self.desVel, self.desAcc = self._vel_sine_sweep(t)
            elif self.xyzType == 102:
                self.desVel = self._vel_square_wave(t)
            elif self.xyzType == 103:
                self.desVel, self.desAcc = self._vel_triangle_wave(t)
            elif self.xyzType == 104:
                self.desVel = self._vel_random_combined(t)
            elif self.xyzType == 105:
                self.desVel, self.desAcc = self._vel_lissajous_3d(t)
            
            # 混合模式
            if self.ctrlType == "xy_vel_z_pos":
                self.desPos[2] = 1.0
                self.desVel[2] = 0
                self.desAcc[2] = 0
        
        # ========== 位置控制模式 ==========
        elif self.ctrlType == "xyz_pos":
            if self.xyzType == 200:
                self._pos_random_step(t)
            elif self.xyzType == 201:
                self._pos_typical_combined(t)
            elif self.xyzType == 202:
                self._pos_circle(t)
            elif self.xyzType == 203:
                self._pos_figure_eight(t)
            elif self.xyzType == 204:
                self._pos_square(t)
            elif self.xyzType == 205:
                self._pos_spiral(t)
            elif self.xyzType == 206:
                self._pos_lissajous_3d(t)
            elif self.xyzType == 207:
                self._pos_random_polynomial(t)
            
            # 航向控制
            if self.yawType == 3:
                self._yaw_follow_velocity(t, Ts, quad)
            elif self.yawType == 0 or self.yawType == 4:
                self.desEul[2] = 0.0
        
        # 组装状态向量
        self.sDes = np.hstack((
            self.desPos, self.desVel, self.desAcc, self.desThr,
            self.desEul, self.desPQR, self.desYawRate
        )).astype(float)
        
        return self.sDes


# ============================================================================
# 可视化与测试
# ============================================================================

class _DummyQuad:
    """测试用四旋翼占位类"""
    def __init__(self):
        self.psi = 0.0
        self.pos = np.zeros(3)


def _simulate_trajectory(ctrl_type, xyz_type, yaw_type, duration, Ts):
    """仿真轨迹生成"""
    quad = _DummyQuad()
    aver_vel = 1 if ctrl_type == "xyz_pos" else 0
    traj_select = np.array([xyz_type, yaw_type, aver_vel])
    traj = Trajectory(quad, ctrl_type, traj_select)

    times = []
    positions = []
    velocities = []
    accelerations = []
    integrated_pos = quad.pos.copy()

    t = 0.0
    while t <= duration + 1e-9:
        sDes = traj.desiredState(t, Ts, quad)
        des_pos = sDes[0:3]
        des_vel = sDes[3:6]
        des_acc = sDes[6:9]
        
        if ctrl_type == "xyz_pos":
            integrated_pos = des_pos.copy()
        else:
            integrated_pos = integrated_pos + des_vel * Ts

        quad.pos = integrated_pos.copy()
        times.append(t)
        positions.append(integrated_pos.copy())
        velocities.append(des_vel.copy())
        accelerations.append(des_acc.copy())
        t += Ts

    return np.array(times), np.vstack(positions), np.vstack(velocities), np.vstack(accelerations)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from datetime import datetime

    # ========== 测试场景配置 ==========
    scenarios = [
        # ===== 新拆分速度轨迹 =====
        {"ctrl": "xyz_vel", "xyz": 1001, "yaw": 3, "duration": 30.0, 
         "label": "Vel: Pitch-Only Steps (NEW)"},
        {"ctrl": "xyz_vel", "xyz": 1002, "yaw": 3, "duration": 30.0, 
         "label": "Vel: Roll-Only Steps (NEW)"},
        {"ctrl": "xyz_vel", "xyz": 1003, "yaw": 3, "duration": 30.0, 
         "label": "Vel: Coupled Steps (NEW)"},
        {"ctrl": "xyz_vel", "xyz": 1011, "yaw": 3, "duration": 30.0, 
         "label": "Vel: Sine Sweep Low-Freq (NEW)"},
        {"ctrl": "xyz_vel", "xyz": 1012, "yaw": 3, "duration": 30.0, 
         "label": "Vel: Sine Sweep High-Freq (NEW)"},
        
        # ===== 原有速度轨迹（对比用） =====
        {"ctrl": "xyz_vel", "xyz": 100, "yaw": 3, "duration": 90.0, 
         "label": "Vel: Pitch-Roll Steps (ORIGINAL 90s)"},
        {"ctrl": "xyz_vel", "xyz": 101, "yaw": 3, "duration": 60.0, 
         "label": "Vel: Sine Sweep (ORIGINAL 60s)"},
        
        # ===== 其他速度轨迹 =====
        {"ctrl": "xyz_vel", "xyz": 102, "yaw": 3, "duration": 30.0, 
         "label": "Vel: Square Wave"},
        {"ctrl": "xyz_vel", "xyz": 103, "yaw": 3, "duration": 30.0, 
         "label": "Vel: Triangle Wave"},
        {"ctrl": "xyz_vel", "xyz": 104, "yaw": 3, "duration": 30.0, 
         "label": "Vel: Random Combined"},
        {"ctrl": "xyz_vel", "xyz": 105, "yaw": 3, "duration": 30.0, 
         "label": "Vel: 3D Lissajous"},
        
        # ===== 位置跟踪轨迹 =====
        {"ctrl": "xyz_pos", "xyz": 200, "yaw": 3, "duration": 60.0, 
         "label": "Pos: Random Steps"},
        {"ctrl": "xyz_pos", "xyz": 201, "yaw": 3, "duration": 45.0, 
         "label": "Pos: Hybrid Demo"},
        {"ctrl": "xyz_pos", "xyz": 202, "yaw": 3, "duration": 30.0, 
         "label": "Pos: Circle"},
        {"ctrl": "xyz_pos", "xyz": 203, "yaw": 3, "duration": 30.0, 
         "label": "Pos: Figure-8"},
        {"ctrl": "xyz_pos", "xyz": 204, "yaw": 3, "duration": 40.0, 
         "label": "Pos: Square"},
        {"ctrl": "xyz_pos", "xyz": 205, "yaw": 3, "duration": 40.0, 
         "label": "Pos: Helix"},
        {"ctrl": "xyz_pos", "xyz": 206, "yaw": 3, "duration": 30.0, 
         "label": "Pos: 3D Lissajous"},
        {"ctrl": "xyz_pos", "xyz": 207, "yaw": 3, "duration": 40.0, 
         "label": "Pos: Random Polynomial"},
    ]

    Ts = 0.05
    
    print("=" * 70)
    print("四旋翼轨迹生成器测试 - 包含新拆分轨迹")
    print("=" * 70)
    print("\n新增轨迹:")
    print("  1001: 俯仰轴阶跃 (30秒)")
    print("  1002: 滚转轴阶跃 (30秒)")
    print("  1003: 俯仰-滚转耦合 (30秒)")
    print("  1011: 正弦扫频低频段 0.01-0.055Hz (30秒)")
    print("  1012: 正弦扫频高频段 0.055-0.1Hz (30秒)")
    print("\n" + "=" * 70 + "\n")
    
    # 生成所有轨迹数据
    sim_results = []
    for i, info in enumerate(scenarios):
        print(f"[{i+1}/{len(scenarios)}] 生成轨迹: {info['label']}")
        times, positions, velocities, accelerations = _simulate_trajectory(
            ctrl_type=info["ctrl"],
            xyz_type=info["xyz"],
            yaw_type=info["yaw"],
            duration=info["duration"],
            Ts=Ts,
        )
        sim_results.append((info, times, positions, velocities, accelerations))
    
    # ========== 绘图 ==========
    print("\n开始绘图...")
    
    # 速度轨迹图
    fig_vel = plt.figure(figsize=(18, 14))
    fig_vel.suptitle("Velocity Tracking Trajectories (Including New Split Trajectories)", 
                     fontsize=16, fontweight='bold')
    
    vel_results = [r for r in sim_results if r[0]["ctrl"] == "xyz_vel"]
    n_vel = len(vel_results)
    
    for i, (info, times, _, velocities, _) in enumerate(vel_results):
        # 速度分量
        ax1 = fig_vel.add_subplot(n_vel, 2, 2*i+1)
        ax1.plot(times, velocities[:, 0], label="Vx", linewidth=1.5, alpha=0.8)
        ax1.plot(times, velocities[:, 1], label="Vy", linewidth=1.5, alpha=0.8)
        ax1.plot(times, velocities[:, 2], label="Vz", linewidth=1.5, alpha=0.8)
        ax1.set_xlabel("Time (s)", fontsize=9)
        ax1.set_ylabel("Velocity (m/s)", fontsize=9)
        ax1.set_title(info["label"], fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=8)
        
        # 速度幅值
        ax2 = fig_vel.add_subplot(n_vel, 2, 2*i+2)
        vel_mag = np.linalg.norm(velocities, axis=1)
        ax2.plot(times, vel_mag, color='purple', linewidth=2, alpha=0.8)
        ax2.set_xlabel("Time (s)", fontsize=9)
        ax2.set_ylabel("|V| (m/s)", fontsize=9)
        ax2.set_title(info["label"] + " - Magnitude", fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 位置轨迹图
    fig_pos = plt.figure(figsize=(16, 12))
    fig_pos.suptitle("Position Tracking Trajectories", fontsize=16, fontweight='bold')
    
    pos_results = [r for r in sim_results if r[0]["ctrl"] == "xyz_pos"]
    n_pos = len(pos_results)
    n_cols = 3
    n_rows = int(np.ceil(n_pos / n_cols))
    
    for i, (info, times, positions, _, _) in enumerate(pos_results):
        ax = fig_pos.add_subplot(n_rows, n_cols, i+1, projection="3d")
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                linewidth=1.5, color='blue', alpha=0.7)
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  c='green', s=100, marker='o', label='Start', alpha=0.9)
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  c='red', s=100, marker='s', label='End', alpha=0.9)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.set_zlabel("Z (m)", fontsize=8)
        ax.set_title(info["label"], fontsize=10, fontweight='bold')
        ax.legend(loc="upper left", fontsize=7)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # 对比图：新拆分轨迹 vs 原始轨迹
    fig_compare = plt.figure(figsize=(16, 10))
    fig_compare.suptitle("Comparison: New Split Trajectories vs Original", 
                        fontsize=16, fontweight='bold')
    
    # 对比1: 阶跃轨迹
    ax1 = fig_compare.add_subplot(2, 2, 1)
    for info, times, _, velocities, _ in vel_results:
        if info["xyz"] in [1001, 1002, 1003]:
            ax1.plot(times, velocities[:, 0], label=f"{info['xyz']}-Vx", linewidth=1.5, alpha=0.7)
            ax1.plot(times, velocities[:, 1], label=f"{info['xyz']}-Vy", linewidth=1.5, alpha=0.7)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title("New Split Step Trajectories (1001/1002/1003)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig_compare.add_subplot(2, 2, 2)
    for info, times, _, velocities, _ in vel_results:
        if info["xyz"] == 100:
            ax2.plot(times, velocities[:, 0], label="100-Vx", linewidth=1.5, alpha=0.7)
            ax2.plot(times, velocities[:, 1], label="100-Vy", linewidth=1.5, alpha=0.7)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_title("Original Combined Trajectory (100)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 对比2: 扫频轨迹
    ax3 = fig_compare.add_subplot(2, 2, 3)
    for info, times, _, velocities, _ in vel_results:
        if info["xyz"] in [1011, 1012]:
            vel_mag = np.linalg.norm(velocities, axis=1)
            ax3.plot(times, vel_mag, label=f"{info['xyz']}", linewidth=2, alpha=0.7)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("|V| (m/s)")
    ax3.set_title("New Split Sweep Trajectories (1011/1012)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig_compare.add_subplot(2, 2, 4)
    for info, times, _, velocities, _ in vel_results:
        if info["xyz"] == 101:
            vel_mag = np.linalg.norm(velocities, axis=1)
            ax4.plot(times, vel_mag, label="101", linewidth=2, alpha=0.7, color='purple')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("|V| (m/s)")
    ax4.set_title("Original Sweep Trajectory (101)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Path("log_trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    vel_path = output_dir / f"velocity_trajectories_split_{timestamp}.png"
    pos_path = output_dir / f"position_trajectories_{timestamp}.png"
    compare_path = output_dir / f"trajectory_comparison_{timestamp}.png"
    
    fig_vel.savefig(vel_path, dpi=200, bbox_inches='tight')
    fig_pos.savefig(pos_path, dpi=200, bbox_inches='tight')
    fig_compare.savefig(compare_path, dpi=200, bbox_inches='tight')
    
    plt.close('all')
    
    print(f"\n✓ 速度轨迹图保存至: {vel_path}")
    print(f"✓ 位置轨迹图保存至: {pos_path}")
    print(f"✓ 对比图保存至: {compare_path}")
    print("\n" + "="*70)
    print("测试完成！新拆分轨迹已验证")
    print("="*70)
